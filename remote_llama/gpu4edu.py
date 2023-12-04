import paramiko
import time

# SSH Connection details
host = 'aurometalsaurus.uvt.nl'
port = 22
username = 'u590531'  # Your username
private_key_path = "C:\\Users\\ooo\\Desktop\\keys\\zan.pem" 
password = 'tesla'  # Your password, if required

# Create an SSH client instance
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the server
private_key = paramiko.Ed25519Key(filename=private_key_path, password=password)
ssh.connect(host, port=port, username=username, pkey=private_key)

# Open a shell
shell = ssh.invoke_shell()

#  sequence of commands for "sepia" node
commands = [
    "srun --nodes=1 --nodelist=lenurple --pty /bin/bash -l",
    "cd /home/u590531/llm/",  # Replace with the correct directory path
    "conda activate hf",  
    "python 7b.py"  
]

# Function to execute commands
def execute_command(cmd, base_timeout=10):
    is_python_script = '.py' in cmd
    timeout = base_timeout * 15 if is_python_script else base_timeout
    shell.send(cmd + "\n")
    
    response = ""
    last_received = time.time()

    while True:
        if shell.recv_ready():
            received_data = shell.recv(1024).decode()
            print(received_data, end='')  # Print data as it comes
            response += received_data
            last_received = time.time()

        # Check if the timeout has elapsed since last data received
        if not is_python_script or (time.time() - last_received > timeout):
            break

        time.sleep(0.6)  # Short sleep to prevent high CPU usage



# Execute each command in the sequence
count=0
for command in commands:
    execute_command(command)
    count+=1

# Close the connection
shell.close()
ssh.close()
