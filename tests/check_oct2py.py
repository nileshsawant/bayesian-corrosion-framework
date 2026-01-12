from oct2py import Oct2Py 
import os
import signal

print("Imported oct2py")
# Octave can be noisy/crashing on exit in some HPC envs, but works for calc.
# We set logger to None to reduce noise if needed.
try:
    o = Oct2Py(executable='octave-cli')
    print("Instantiated Oct2Py")
    res = o.eval("1+1")
    print(f"Result: {res}")
except Exception as e:
    print(f"Error: {e}")

