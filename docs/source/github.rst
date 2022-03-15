GitHub
======

How to collaborate to a repository such as `EasyHPC <https://github.com/pescap/EasyHPC>`_? If you want to send Pull Requests to open source directories, you have to follow these steps:

Create an issue
---------------
To propose changes or enhancements to the code, it is preferable to use the Issues section.

- Go to the `issues <https://github.com/pescap/EasyHPC/issues>`_ section of the GitHub repository.
- Click on ``New issue`` button.
- Write a **title** for the issue and in the **write** section describe it.
- Add **labels** and **assignees**, you can do this in the right panel when creating an issue.

With this, you successfully added a new issue that every other collaborator can see and comment.


Fork the repository to your GitHub account
------------------------------------------

- Go to the EasyHPC GitHub directory `(click here) <https://github.com/pescap/EasyHPC>`_.
- Click on the ``Fork`` button on the top-right hand corner of the window.
- Choose where you want to fork EasyHPC.
  
Work locally on the forked repository
-------------------------------------
Now that you have forked the repository, you will clone it locally on your computer.

- Go to the forked GitHub directory webpage. It should be something like: ::

	https://github.com/your_username/EasyHPC/

- Click on the ``clone`` button and copy the directory URL.
  
- In your terminal type(This step just needs to be done when you don´t have the local repo of the forked repository)::

	$ git clone directory_url

- Go to the directory and create your own branch. For example, assume that you name this branch ``neo``::
  
	$ git branch neo

- Switch to branch ``neo``::
  
  	$ git checkout neo

Now, you are ready to work on this branch, to make all the changes that you want to the code.  

How to keep your local repository up to date
--------------------------------------------

Configure a remote:
   
1. List the current remote repository for your fork: ::

	$ git remote -v
	> origin https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
	> origin https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)

2. Specify a new remote upstream: ::

	$ git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git

3. Verify the new upstream: ::

	$ git remote -v
	> origin   https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
	> origin   https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
	> upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (fetch)
	> upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (push)

Sync your forked repository with the original (Forked repo. webpage).

1. Go to the main page of your forked repository.
2. **(optional)** Click on the ``main`` button to change the **branch** to the one you are working with.
3. Click on the ``Fetch upstream`` button.
4. If you want to see and compare which changes have been made since your last pull, you can click on ``Compare``. Then click on ``Fetch and merge``.

Then, you can either **fetch** your forked repo into the local repo using: ::

$ git fetch origin <branch_name>

Or, you can **Pull** changes from your **forked** repository: ::

$ git pull

In case you haven't synced your forked repository, you can do this: ::

$ git pull upstream

**Warning**: You will lose your work in the working space if you pull any repository before you commit into the local repo.

Push to the main/forked repository
----------------------------------	

Remember that if you added files to your directory in the working space, those files must be added before commiting: ::

$ git add <file_name>

Once your changes are done, you can commit and push them to the remote branch ``neo``: ::

$ git commit -a -m "message about what you added"
$ git push origin neo 

If you want to merge your changes to the original ``EasyHPC``, go back to your forked page, e.g.: ::

 https://github.com/your_username/EasyHPC/

Check compatibility and propose a Pull Request. 

**Note**: Before you submit a pull request: 

- Verify that your forked version is up to date with the original one.
- Remember to apply `black <https://pypi.org/project/black/>`_ to your Python code. Black allows to format Python code. To install Black and apply it to a ``my_code.py`` script: ::

   	$ pip install black
 	$ black my_code.py

You're ready to collaborate to any Open-Source repository on GitHub!

How to manually link an issue with a pull request
-------------------------------------------------

1. On the upstream GitHub repository click on ``Pull requests``.
2. Click on the pull request that you would like to link to an issue.
3. In the right panel, ``Development`` section click |:gear:|.
4. Select the issue you want to link.

**Note**: You can do this every time you are about to present a pull request to the upstream repository.

Milestones
----------
To better manage/see due dates, completion percentage, open/closed issues and pull requests associated with a specific part/characteristic of the project. 

1. Go to the main page of the original repository.
2. Click on ``Issues`` or ``Pull requests``.
3. Next to the ``Labels`` button, click ``Milestones``.
4. You can either **create** a milestone or edit an existing milestone.
5. Type the milestone's title and description.

**Note**: When you delete milestones, issues and pull requests are not affected.