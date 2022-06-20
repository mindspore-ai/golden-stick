# Security for MindSpore Golden Stick

## Security Risk Description

When MindSpore Golden Stick is used to optimize AI model, if the user-defined network model (for example, Python code for generating the MindSpore computational graph) is provided by an untrusted third party, malicious code may exist and will be loaded and executed to attack the system.

## Security Usage Suggestions

1. Run MindSpore Golden Stick in the sandbox.
2. Run MindSpore Golden Stick as a non-root user.
3. Ensure that the source of a network model is trustworthy or enter secure network model parameters to prevent model parameters from being tampered with.
