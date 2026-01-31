# Specify the filename
filename = "my_new_file.txt"

# Open the file in 'write' mode ('w')
# If the file doesn't exist, it will be created.
# If the file exists, its contents will be truncated (emptied).
try:
    with open(filename, 'w') as file:
        # Write some content to the file
        file.write("Hello, this is a new text file.\n")
        file.write("This is the second line.\n")
        file.write("You can add as many lines as you want.\n")
    print(f"File '{filename}' created and content written successfully.")

except IOError as e:
    print(f"Error: Could not create or write to file '{filename}'. {e}")