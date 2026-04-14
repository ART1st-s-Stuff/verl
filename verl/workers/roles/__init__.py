# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module

__all__ = ["CriticWorker", "ActorWorker", "RewardModelWorker"]


def __getattr__(name):
    if name == "ActorWorker":
        return import_module(".actor", __name__).ActorWorker
    if name == "CriticWorker":
        return import_module(".critic", __name__).CriticWorker
    if name == "RewardModelWorker":
        try:
            return import_module(".reward_model", __name__).RewardModelWorker
        except ImportError:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
