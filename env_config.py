import subprocess
import sys

def run_shell_script(script_name):
    """Run a shell script and handle errors."""
    result = subprocess.run([script_name], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running script: {script_name}")
        print(result.stderr.decode())
        sys.exit(result.returncode)
    print(result.stdout.decode())

def main():
    script_name = "./setup_env.sh"
    run_shell_script(script_name)

if __name__ == "__main__":
    main()

