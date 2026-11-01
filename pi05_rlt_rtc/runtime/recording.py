"""Bounded asynchronous archive of exact planner inputs (not continuous video).

Disk I/O and image encoding stay off the controller thread. Overflow and write
errors fail the trial; they never silently drop observations. Files contain only
JSON, numeric arrays and lossless RGB bytes, never pickled Python objects.
"""
from __future__ import annotations

import dataclasses
import json
import queue
import threading
from pathlib import Path

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment.rpc import encode_observation, json_value

FORMAT = 'pi05-planner-request-v1'


def write_request(path, request, observation, pending_prefix, metadata):
    payload = {'format': FORMAT, 'metadata': metadata,
               'request': dataclasses.asdict(request),
               'observation': encode_observation(observation),
               'pending_prefix': pending_prefix}
    path = Path(path)
    temporary = path.with_suffix('.partial')
    try:
        with temporary.open('x') as stream:
            json.dump(payload, stream, default=json_value, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_request(path):
    from pi05_rlt_rtc.contracts.spaces import HorizonContract
    from pi05_rlt_rtc.tools.planner_server import _observation_from_json, _request_from_json
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ContractError('recording exceeds the 16 MiB request limit')
    data = json.loads(path.read_text())
    if data.get('format') != FORMAT:
        raise ContractError('unsupported recording format')
    contract = HorizonContract(**data['metadata']['runtime_contract']['horizon'])
    request = _request_from_json(data['request'], contract)
    obs = _observation_from_json(data['observation'])
    request.validate(contract, allow_warm_start=True)
    obs.validate(proprio_dim=9)
    if (obs.env_id, obs.episode_id, obs.tick) != (request.env_id, request.episode_id, request.capture_tick):
        raise ContractError('recorded request/observation identity mismatch')
    prefix = np.asarray(data['pending_prefix'], np.float32)
    if prefix.shape != (contract.delay, 8) or not np.isfinite(prefix).all() or np.any(np.abs(prefix) > 1):
        raise ContractError('invalid recorded pending prefix')
    return {**data, 'request': request, 'observation': obs, 'pending_prefix': prefix}


class AsyncRequestRecorder:
    def __init__(self, directory, metadata, *, capacity=16):
        if type(capacity) is not int or capacity < 1:
            raise ContractError('recording capacity must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self._queue = queue.Queue(maxsize=capacity)
        self._error = None
        self._closed = False
        self._count = 0
        self._thread = threading.Thread(target=self._write, name='pi05-observation-writer', daemon=True)
        self._thread.start()

    def check(self):
        if self._error is not None:
            raise ContractError(f'observation recording failed: {self._error}')

    def submit(self, request, observation, pending_prefix):
        self.check()
        if self._closed:
            raise ContractError('observation recorder is closed')
        item = (self.directory / f'request-{self._count:06d}.json', request,
                observation, np.array(pending_prefix, copy=True), self.metadata)
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            self._error = 'bounded observation queue is full'
            raise ContractError(self._error) from exc
        self._count += 1

    def _write(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                write_request(*item)
            except Exception as exc:
                self._error = f'{type(exc).__name__}: {exc}'
            finally:
                self._queue.task_done()

    def close(self):
        # Call after stopping control. Draining disk writes must never delay a stop.
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        self.check()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
