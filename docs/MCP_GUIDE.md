# MCP (Model Context Protocol) Guide

This guide explains how to use MCP servers with Claude Code to enhance your development experience.

## What is MCP?

Model Context Protocol (MCP) allows Claude Code to access external tools and services, extending its capabilities beyond basic file operations. The Golomb project is configured with four MCP servers to improve your workflow.

## Configured MCP Servers

### 1. GitHub Server (`@modelcontextprotocol/server-github`)

**Capabilities:**
- Create and manage issues
- Create and review pull requests
- List and search repositories
- Manage GitHub Actions workflows
- Access repository metadata

**Setup:**
1. Create a GitHub Personal Access Token (PAT):
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo`, `workflow`, `read:org`
   - Copy the generated token

2. Add the token to your MCP configuration:
   - Open [.claude/mcp.json](../.claude/mcp.json)
   - Replace the empty `GITHUB_PERSONAL_ACCESS_TOKEN` value with your token
   - Save the file

**Example usage:**
```
Ask Claude: "Create an issue to track the performance regression in greedy_seed"
Ask Claude: "List all open pull requests for this repository"
```

### 2. Git Server (`@modelcontextprotocol/server-git`)

**Capabilities:**
- Advanced git operations (log, diff, blame)
- Commit history analysis
- Branch management
- Detailed diff viewing
- Commit searching

**Setup:**
No additional configuration required. Works automatically with your local git repository.

**Example usage:**
```
Ask Claude: "Show me the git log for the last 10 commits"
Ask Claude: "What changes were made to golomb.cpp in the last commit?"
Ask Claude: "Who last modified the MCTS implementation?"
```

### 3. Filesystem Server (`@modelcontextprotocol/server-filesystem`)

**Capabilities:**
- Extended file system access
- Recursive directory traversal
- File watching for changes
- Advanced file search
- Batch file operations

**Setup:**
Pre-configured to access: `c:\Users\nicol\Documents\Projet\Golomb`

**Security Note:** The filesystem server only has access to the Golomb project directory.

**Example usage:**
```
Ask Claude: "Find all files containing 'MCTS' in their name"
Ask Claude: "List all C++ source files modified in the last 7 days"
```

### 4. Memory Server (`@modelcontextprotocol/server-memory`)

**Capabilities:**
- Store notes and context between Claude sessions
- Remember project-specific information
- Track ongoing tasks and decisions
- Maintain knowledge base

**Setup:**
No additional configuration required. Memory is stored locally.

**Example usage:**
```
Ask Claude: "Remember that we decided to use PUCT=1.4 for MCTS"
Ask Claude: "What did we discuss about the evolutionary algorithm?"
```

## Prerequisites

All MCP servers are run via `npx` (no installation required), but you need:
- **Node.js 18+** installed
- **npm** available in your PATH

Check if you have them:
```bash
node --version
npm --version
```

If not installed, download from: https://nodejs.org/

## Enabling MCP in Claude Code

MCP configuration is automatically loaded from [.claude/mcp.json](../.claude/mcp.json).

To verify MCP is working:
1. Restart VS Code after editing `.claude/mcp.json`
2. Ask Claude: "What MCP servers are available?"
3. Claude should list the 4 configured servers

## Troubleshooting

### MCP servers not loading
- Ensure Node.js 18+ is installed
- Check that `npx` is in your PATH
- Restart VS Code
- Check VS Code Developer Tools (Help → Toggle Developer Tools) for errors

### GitHub token not working
- Verify the token has correct scopes (`repo`, `workflow`, `read:org`)
- Ensure the token hasn't expired
- Check that the token is properly set in `.claude/mcp.json`

### Permission errors with filesystem server
- Verify the path in `.claude/mcp.json` points to your project directory
- Ensure you have read/write permissions for the directory

## Best Practices

1. **GitHub Token Security:**
   - Never commit your token to git
   - Add `.claude/mcp.json` to `.gitignore` if it contains sensitive tokens
   - Rotate tokens periodically

2. **Using Memory Server:**
   - Store important architectural decisions
   - Keep track of performance benchmarks findings
   - Document workarounds for tricky bugs

3. **Filesystem Operations:**
   - Be specific about file paths when asking Claude
   - Use relative paths from project root

## Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [Claude Code Documentation](https://docs.claude.com/claude-code)

## Need Help?

If you encounter issues with MCP:
1. Check the troubleshooting section above
2. Review Claude Code logs in VS Code
3. Open an issue on the [Claude Code GitHub](https://github.com/anthropics/claude-code/issues)
