#!/usr/bin/env bash

# Example usage:
#   bash create_env.sh --name "myenv"
# If --name is not provided, the default environment name is "venv".

ENV_NAME="venv"

# Parse command line argument
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --name)
            ENV_NAME="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Creating virtual environment: ${ENV_NAME} ..."
python3 -m venv "${ENV_NAME}"
echo "Activating virtual environment ..."
source "${ENV_NAME}/bin/activate"
echo "Done."
