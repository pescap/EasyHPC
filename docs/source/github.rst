======
GitHub
======

How to collaborate to a repository such as `EasyHPC <https://github.com/pescap/EasyHPC>`_? If you want to send Pull Requests to open source repositories, you are encouraged to follow these steps:

Create an issue
---------------
To propose changes or enhancements to the code, it is preferable to use the Issues section.

- Go to the `issues <https://github.com/pescap/EasyHPC/issues>`_ section of the GitHub repository.
- Click on ``New issue`` button.
- Define a **title** for the issue fill in the **write** section.
- Add **labels** and **assignees** (right panel).

With this, you successfully created a new issue that every other collaborator can see and comment.


Fork the repository to your GitHub account
------------------------------------------

This step creates your own remote copy of the repo you want to work on. This way you can modify the code, create your own branches and keep the main code of the main repo clean and safe before merging your changes.

- Go to the EasyHPC GitHub directory (`click here <https://github.com/pescap/EasyHPC>`_).
- Click on the ``Fork`` button on the top-right hand corner of the window.
- Choose where you want to fork EasyHPC.


SSH key configuration
---------------------

You can log into GitHub using the Secure Shell (SSH) protocol.

If you don't have an SSH key, use the following command to create one::

	 $ ssh-keygen

You can view your SSH public key by running: ::

	 $ cat ~/.ssh/id_rsa.pb

Add the SSH key to your GitHub profile. Copy and paste it here:

* Github > Settings > SSH and GPG Keys > New SSH key.

You can also refer to this link for detailed instructions: ::

	https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux

Warning: Don’t forget to add your SSH private key to the `ssh-agent`!

Workflow
--------
The next series of steps will help you understand the git workflow and how to either update your repo or commiting your changes to then push to your forked repo and send pull requests to add your contributions.

Cloning your forked repository
********************************
Now that you have forked the repository, you will clone it locally on your computer to now generate a copy of it into your disk.

- Go to the forked GitHub directory webpage. It should be something like: ::

	https://github.com/your_username/EasyHPC/

- Click on the ``clone`` button and copy the directory URL.
  
- In your terminal type (This step just needs to be done when you don't have the local repo of the forked repository)::

	$ git clone directory_url

- Go to the directory and create your own branch. For example, assume that you name this branch ``neo``::
  
	$ git branch neo

- Switch to branch ``neo``::
  
  	$ git checkout neo

Now, you are ready to work on this branch, to make all the changes that you want to the code.  

How to keep your local repository up to date or git pull
**********************************************************

Once you're done with the ssh key configuration, let's set up a remote:
   
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

Commit and push to the main/forked repository
***********************************************	

Remember that if you added files to your directory in the working space, those files must be added before commiting: ::

$ git add <file_name>

Once your changes are done, you can commit and push them to the remote branch ``neo``, note that when you git commit you're saving those changes in your local repo and then git push uploads your local repo into either your origin(highly recomended) or upstream remote repo: ::

	$ git commit -a -m "message about what you added"
	$ git push origin neo 

Notice that you can link the pull request to an issue using a keyword (see `here <https://docs.github.com/es/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`_).

For example, if you commit solves issue number `#90`, you can run::

	$ git commit -a -m "fixes #90"
	$ git push origin neo

This will automatically close issue `#90`.

Pull requests for merging your changes into the original repo
***************************************************************

If you want to merge your changes to the original ``EasyHPC``, go back to your forked page, e.g.: ::

 https://github.com/your_username/EasyHPC/

Check compatibility and propose a Pull Request. You should see your pushes on the github website of your fork and it will suggest you to send a pull request 

**Note**: Before you submit a pull request: 

- Verify that your forked version is up to date with the original one.
- Remember to apply `black <https://pypi.org/project/black/>`_ to your Python code. Black allows to format Python code. To install Black and apply it to a ``my_code.py`` script: ::

   	$ pip install black
 	$ black my_code.py

You're ready to collaborate to any Open-Source repository on GitHub!

Extra help
----------

How to manually link an issue with a pull request
***************************************************

1. On the upstream GitHub repository click on ``Pull requests``.
2. Click on the pull request that you would like to link to an issue.
3. In the right panel, ``Development`` section click on the gear emoji.
4. Select the issue you want to link.

**Note**: You can do this every time you are about to present a pull request to the upstream repository.

Milestones
************
To manage better due dates, completion percentage, open/closed issues and pull requests associated with a specific part/characteristic of the project:

1. Go to the main page of the original repository.
2. Click on ``Issues`` or ``Pull requests``.
3. Next to the ``Labels`` button, click ``Milestones``.
4. You can either **create** a milestone or edit an existing milestone.
5. Type the milestone's title and description.

**Note**: When you delete milestones, issues and pull requests are not affected.

GitHub Actions
****************
`GitHub Actions <https://github.com/features/actions>`_ allows to automate workflows. They can be accessed via the ``Actions`` in the home GitHub repository (web).

Workflows are stored in `.github/workflow <https://github.com/pescap/EasyHPC/tree/main/.github/workflows>`_. A simple workflow was created in `issues.yml <https://github.com/pescap/EasyHPC/blob/main/.github/workflows/issues.yml>`_. It follows the general structure for workflows: ::

	name: Close inactive issues #name for the workflow
	on: #when it is runned. It can be on schedule or via a manual trigger
	  schedule:
	    - cron: "30 1 * * *" #here, it runs every day

	jobs: # each workflow in subdivised into jobs
	  close-issues: #here, one job called close-issues
	    runs-on: ubuntu-latest #on which machine it is runned
	    permissions: #the permissiones for the workflow
	      issues: write
	      pull-requests: write
	    steps:
	      - uses: actions/stale@v3
	        with:
	          days-before-issue-stale: 7
	          days-before-issue-close: 7
	          stale-issue-label: "stale"
	          stale-issue-message: "This issue is stale because it has been open for 7 days with no activity."
	          close-issue-message: "This issue was closed because it has been inactive for 7 days since being marked as stale."
	          days-before-pr-stale: -1
	          days-before-pr-close: -1
	          repo-token: ${{ secrets.GITHUB_TOKEN }}

This workflow stales inactive issues after 7 days, and closes them 7 days later. The code is issued from `this link <https://docs.github.com/en/github-ae@latest/actions/managing-issues-and-pull-requests/closing-inactive-issues>`_.
