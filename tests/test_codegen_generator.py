from backend.codegen.generator import ProgramGenerator


class StubLLMClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict] = []

    def chat_completion(self, messages, temperature=0.3, max_tokens=None, response_format=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        return self.responses.pop(0)


def test_generate_recovers_missing_required_files_via_followup_completion():
    llm = StubLLMClient(
        responses=[
            """```python model.py
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def get_search_space(cls):
        return {}
```

```python loss.py
import torch
import torch.nn as nn


class Loss(nn.Module):
    def forward(self, pred, target, **kwargs):
        return ((pred - target) ** 2).mean()
```
""",
            """### data_pipeline.py
```python
from torch.utils.data import DataLoader, TensorDataset
import torch


class DataModule:
    def setup(self, stage=None):
        features = torch.randn(8, 4)
        targets = torch.randn(8, 1)
        self.train_dataset = TensorDataset(features, targets)
        self.val_dataset = TensorDataset(features, targets)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=4)
```

### train_loop.py
```python
class Trainer:
    def fit(self, model, loss_fn, datamodule):
        datamodule.setup("fit")
        return {"status": "ok"}

    def validate(self, model, loss_fn, datamodule):
        return {"val_loss": 0.0}

    def test(self, model, loss_fn, datamodule):
        return {"test_loss": 0.0}
```
""",
        ]
    )

    generator = ProgramGenerator(llm_client=llm)
    program = generator.generate("生成一个最小可运行训练程序")

    assert program.model_code.strip()
    assert program.loss_code.strip()
    assert program.data_pipeline_code.strip()
    assert program.train_loop_code.strip()
    assert len(llm.calls) == 2
    assert llm.calls[0]["max_tokens"] == 8000
    assert llm.calls[1]["max_tokens"] == 6000


def test_fix_code_preserves_existing_files_when_llm_omits_them():
    llm = StubLLMClient(
        responses=[
            """### model.py
```python
class Model:
    pass
```
"""
        ]
    )
    generator = ProgramGenerator(llm_client=llm)
    program = generator._parse_response(
        """```python model.py
class ExistingModel:
    pass
```
```python loss.py
class ExistingLoss:
    pass
```
```python data_pipeline.py
class ExistingDataModule:
    pass
```
```python train_loop.py
class ExistingTrainer:
    pass
```
""",
        "intent",
    )

    repaired = generator.fix_code(program, [{"file": "model.py", "message": "dummy"}], "static_analysis")

    assert "class Model" in repaired.model_code
    assert "class ExistingLoss" in repaired.loss_code
    assert "class ExistingDataModule" in repaired.data_pipeline_code
    assert "class ExistingTrainer" in repaired.train_loop_code
