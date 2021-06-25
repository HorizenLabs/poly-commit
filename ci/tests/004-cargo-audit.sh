#!/bin/bash

set -eo pipefail

cargo audit --json > audit.json
cat audit.json | jq .

VULNERABILITIES=$(jq .vulnerabilities.found audit.json)

if [ $VULNERABILITES==false ]
then
echo "No vulnerabilities found"
else
exit 1
fi

