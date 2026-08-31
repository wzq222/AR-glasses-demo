# ONNX Runtime 运行时不做混淆（保持模型会话正常）
-keep class ai.onnxruntime.** { *; }

# PyTorch Mobile（TorchScript .pt 直接加载）
-keep class org.pytorch.** { *; }