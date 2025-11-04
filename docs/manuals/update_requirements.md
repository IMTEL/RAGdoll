# Steps to Update Dependencies

By following these steps, you maintain an accurate and up-to-date dependency list in your `pyproject.toml` file.

**1. Create a Virtual Environment & Install Dependencies**

Run the following command to create a virtual environment (stored in the `.venv` directory):
```bash
uv sync
```

**2. Activate the Virtual Environment**

The virtual environment is always active when using `uv` commands. However, if you need to manually activate it in your terminal, do so as follows:

*On Windows:*
```bash
source .venv\Scripts\activate
```

*On Linux and macOS:*
```bash
source .venv/bin/activate
```

**4. Add New Dependencies**

If you've added new imports to your code that require additional packages, add them either by command:
```bash
uv add <package-name>
```
or by manually adding them to the `pyproject.toml` file under the `[project.dependencies]` section. Specify the package name and version as needed.

For example:
```toml
[project]
dependencies = [
    "new-package==1.2.3",
    # ...existing dependencies...
]
```

After updating the `pyproject.toml` file run `uv sync` again
This ensures that the new dependencies are installed and locked.

**5. Update the Lock File**

Push the changes to the lock file to ensure consistency.

**6. Deactivate the Virtual Environment**

Once you've updated the dependencies, deactivate the virtual environment:
<pre>deactivate</pre>
