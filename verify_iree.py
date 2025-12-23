import iree.compiler
import iree.runtime

print("IREE modules imported successfully.")
try:
    print(f"IREE Compiler package info: {iree.compiler.__package__}")
    print(f"IREE Runtime package info: {iree.runtime.__package__}")
except Exception as e:
    print(f"Error accessing package info: {e}")
