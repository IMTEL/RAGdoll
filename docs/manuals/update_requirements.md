
# Steps to Update requirements.txt

By following these steps, you maintain an accurate and up-to-date requirements.txt file that reflects your project's dependencies. This practice enhances collaboration, simplifies the setup process for new developers, and ensures smooth deployment across different environments.

**1. Create a Virtual Environment**


Run the following command to create a virtual environment named venv:
<pre> python -m venv venv </pre>


**2. Activate the Virtual Environment**

Before installing or updating packages, activate the virtual environment.

*On windows:*
<pre>source venv/Scripts/activate</pre>

*On linux and mac:*
<pre>source venv/bin/activate</pre>



**3. Install Existing Dependencies**

Install all the packages listed in the current requirements.txt file:
<pre>pip install -r requirements.txt</pre>

**4. Install New Packages**

If you've added new imports to your code that require additional packages, install them using pip.

For each new package, run:
<pre>pip install [package]</pre>
Replace package_name with the name of the package you need.

5. Update the requirements.txt File

After installing the new packages, update the requirements.txt file to include all currently installed packages and their versions:
<pre>pip freeze > requirements.txt
</pre>
This command overwrites the existing requirements.txt file with an updated list of all packages installed in the virtual environment.

*6. Deactivate the Virtual Environment**

Once you've updated the requirements.txt file, deactivate the virtual environment:
<pre>
deactivate
</pre>

You can also opt to kill your terminal to deactivate the virtual enviroment. 



