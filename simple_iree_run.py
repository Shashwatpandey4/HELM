import iree.compiler
import iree.runtime
import numpy as np

# 1. Define the MLIR module (simple multiply function)
# This is a textual representation of the program we want to run.
mlir_code = """
module {
  func.func @multiply(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

print("Compiling MLIR...")
# 2. Compile the MLIR to a VM module (bytecode)
# We target the 'vmvx' backend which is a reference CPU interpreter,
# good for simple validation without needing specific hardware drivers.
compiled_module = iree.compiler.compile_str(
    mlir_code, 
    target_backends=["vmvx"]
)

print("Loading module...")
# 3. Load the compiled module into the runtime
config = iree.runtime.Config("local-task")
vm_module = iree.runtime.VmModule.from_flatbuffer(config.vm_instance, compiled_module)
ctx = iree.runtime.SystemContext(config=config)
ctx.add_vm_module(vm_module)

# 4. Invoke the function
print("Running 'multiply' function...")
arg0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
arg1 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

# Get the function from the loaded module context
# The MLIR module was anonymous, so it defaults to 'module' usually.
# However, we can inspect ctx.modules
f = ctx.modules.module["multiply"]
result = f(arg0, arg1)

print("\nResults:")
print(f"Input A: {arg0}")
print(f"Input B: {arg1}")
print(f"Output : {result.to_host()}")
