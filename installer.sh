#!/usr/bin/env bash

# Example usage:
#   bash installer.sh --filename "requirements.txt"
# If --filename is not provided, the default is "requirements.txt".

REQUIREMENTS_FILE="requirements.txt"

# Parse command line argument
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --filename)
            REQUIREMENTS_FILE="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: Requirements file '${REQUIREMENTS_FILE}' not found!"
  exit 1
fi

echo "Installing libraries from ${REQUIREMENTS_FILE} ..."
pip install -r "${REQUIREMENTS_FILE}"
echo "Done."
