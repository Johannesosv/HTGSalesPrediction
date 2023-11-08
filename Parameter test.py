import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python myscript.py <argument>")
    sys.exit(1)

# Access the argument
argument = sys.argv[1]
print("The provided argument is:", argument)
