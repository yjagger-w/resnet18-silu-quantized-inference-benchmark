\# ResNet18-SiLU Quantized Inference Benchmark



A minimal benchmark project for evaluating a ResNet18-SiLU model with PyTorch, ONNX, and ONNX Runtime.



\## Current Baseline



\- Model: ResNet18-SiLU

\- Dataset: CIFAR-10 test set

\- Checkpoint: `resnet18\_cifar10.pth`

\- Test samples: 10,000

\- Correct predictions: 9,374

\- FP32 accuracy: 93.7400%



\## Run FP32 Evaluation



```bash

python scripts/evaluate\_fp32.py --checkpoint checkpoints/resnet18\_cifar10.pth --data-root data --batch-size 128 --device cpu



Expected Output

Missing keys: 0

Unexpected keys: 0

Input shape: (128, 3, 32, 32)

Output shape: (128, 10)

Correct: 9374

Total: 10000

Accuracy: 93.7400%



Roadmap

&#x20;Reproduce PyTorch FP32 baseline

&#x20;Export FP32 model to ONNX

&#x20;Validate PyTorch and ONNX Runtime numerical consistency

&#x20;Run ONNX Runtime FP32 benchmark

&#x20;Apply ONNX Runtime static INT8 quantization

&#x20;Compare accuracy, model size, latency, and throughput

