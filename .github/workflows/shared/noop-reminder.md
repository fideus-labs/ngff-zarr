**Important**: If no action is needed after completing your analysis, you **MUST** call the `noop` safe-output tool with a brief explanation. Failing to call any safe-output tool is the most common cause of safe-output workflow failures.

Call the safe-outputs CLI directly as a standalone command. Do not prepend
`command -v`, chain it with other commands, or request interactive approval.

```bash
safeoutputs noop '{"message":"No action needed: [brief explanation of what was analyzed and why]"}'
```
