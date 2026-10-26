======
GitHub
======

This page explains how to contribute to a repository such as `EasyHPC <https://github.com/pescap/EasyHPC>`__. The same steps work for any open-source project on GitHub.


Create an issue
---------------

Before changing the code, describe the problem or the improvement in an issue, so that other collaborators can discuss it.

- Go to the `issues <https://github.com/pescap/EasyHPC/issues>`__ section of the repository.
- Click on the ``New issue`` button.
- Write a title and a description of the issue.
- Add labels and assignees in the right panel.

Every collaborator can now see and comment on the issue.


Fork the repository
-------------------

A fork is your own copy of the repository on GitHub. You can create branches and push to it freely, and the original repository stays clean until your changes are reviewed and merged.

- Go to the `EasyHPC repository <https://github.com/pescap/EasyHPC>`__.
- Click on the ``Fork`` button in the top-right corner.
- Choose the account where you want to create the fork.


Configure an SSH key
--------------------

GitHub authenticates your pushes with an SSH key. If you do not have one yet, create it::

    $ ssh-keygen -t ed25519 -C "your_email@example.com"

Load the key into the SSH agent, so that you type its passphrase only once per session::

    $ eval "$(ssh-agent -s)"
    $ ssh-add ~/.ssh/id_ed25519

Print the public key and copy it::

    $ cat ~/.ssh/id_ed25519.pub

On GitHub, go to ``Settings`` > ``SSH and GPG keys`` > ``New SSH key`` and paste it. Then check the connection::

    $ ssh -T git@github.com

The `GitHub documentation <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`__ covers the details for each operating system.

.. warning::

    Never share or upload the private key ``~/.ssh/id_ed25519``. Only the ``.pub`` file goes to GitHub.


Workflow
--------

The following steps take you from a fork to a pull request.


Clone your fork
***************

Go to your fork on GitHub, for example ``https://github.com/your_username/EasyHPC``, click on the ``Code`` button and copy the SSH URL. Then clone it and enter the directory::

    $ git clone git@github.com:your_username/EasyHPC.git
    $ cd EasyHPC

You only need to clone once. Create a branch for your work, for example ``neo``, and switch to it::

    $ git switch -c neo

You are ready to edit the code on this branch.


Keep your fork up to date
*************************

Your clone knows your fork as ``origin``. Add the original repository as a second remote called ``upstream``::

    $ git remote add upstream git@github.com:pescap/EasyHPC.git
    $ git remote -v
    origin    git@github.com:your_username/EasyHPC.git (fetch)
    origin    git@github.com:your_username/EasyHPC.git (push)
    upstream  git@github.com:pescap/EasyHPC.git (fetch)
    upstream  git@github.com:pescap/EasyHPC.git (push)

To bring the latest changes of the original repository into your ``main`` branch and your fork::

    $ git switch main
    $ git pull upstream main
    $ git push origin main

You can also click on ``Sync fork`` on the GitHub page of your fork, and then run ``git pull`` locally.

To bring those changes into your working branch::

    $ git switch neo
    $ git merge main

If you have uncommitted changes that conflict with the incoming ones, Git refuses to pull or merge and leaves your files untouched. Commit your changes first, or set them aside with ``git stash`` and recover them with ``git stash pop``.


Commit and push
***************

Add new or modified files to the next commit, then commit them::

    $ git add my_file.py
    $ git commit -m "Describe what you changed"

A commit only saves the changes in your local repository. To upload them to your fork, push the branch::

    $ git push origin neo

Push to your fork (``origin``). Changes reach the original repository through a pull request.

If your commit solves an issue, mention it with a keyword in the commit message or in the pull request description, for example::

    $ git commit -m "Fix the SSH instructions, fixes #90"

GitHub closes issue ``#90`` when the pull request is merged. See `Linking a pull request to an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue>`__.


Send a pull request
*******************

Go to your fork on GitHub. After a push, GitHub shows a ``Compare & pull request`` button. Click on it, check that the base repository is ``pescap/EasyHPC`` and the base branch is ``main``, describe your changes and create the pull request.

Before you send it:

- Make sure your branch includes the latest changes of the original repository.
- Format your Python code with `black <https://pypi.org/project/black/>`__::

    $ pip install black
    $ black my_code.py

You are ready to contribute to any open-source repository on GitHub.


Extra help
----------

Link an issue to a pull request manually
****************************************

#. On the original repository, click on ``Pull requests``.
#. Open the pull request you want to link.
#. In the ``Development`` section of the right panel, click on the gear icon.
#. Select the issue.


Milestones
**********

Milestones group issues and pull requests around a goal, with a due date and a completion percentage.

#. Go to the main page of the original repository.
#. Click on ``Issues`` or ``Pull requests``.
#. Next to the ``Labels`` button, click on ``Milestones``.
#. Create a milestone or edit an existing one, with a title and a description.

Deleting a milestone does not affect its issues and pull requests.


GitHub Actions
**************

`GitHub Actions <https://docs.github.com/en/actions>`__ automates workflows. They are listed in the ``Actions`` tab of the repository and stored in `.github/workflows <https://github.com/pescap/EasyHPC/tree/main/.github/workflows>`__.

The workflow `issues.yml <https://github.com/pescap/EasyHPC/blob/main/.github/workflows/issues.yml>`__ closes inactive issues. It follows the general structure of a workflow:

.. code-block:: yaml

    name: Close inactive issues
    on:
      schedule:
        - cron: "30 1 * * *"

    jobs:
      close-issues:
        runs-on: ubuntu-latest
        permissions:
          issues: write
          pull-requests: write
        steps:
          - uses: actions/stale@v11
            with:
              days-before-issue-stale: 60
              days-before-issue-close: 60
              stale-issue-label: "stale"
              stale-issue-message: "This issue is stale because it has been open for 60 days with no activity."
              close-issue-message: "This issue was closed because it has been inactive for 60 days since being marked as stale."
              days-before-pr-stale: -1
              days-before-pr-close: -1

- ``name`` is the name shown in the ``Actions`` tab.
- ``on`` says when the workflow runs. Here it runs every day at 01:30 UTC; it can also run on every push, on every pull request or by hand.
- ``jobs`` lists the jobs of the workflow. Each job runs on a fresh machine given by ``runs-on``.
- ``permissions`` grants the job the right to edit issues and pull requests.
- ``steps`` lists what the job does. Here it uses the ready-made action `actions/stale <https://github.com/actions/stale>`__.

The workflow marks an issue as stale after 60 days without activity and closes it 60 days later. Pull requests are never marked or closed. It is adapted from `Closing inactive issues <https://docs.github.com/en/actions/tutorials/manage-your-work/close-inactive-issues>`__ in the GitHub documentation.
