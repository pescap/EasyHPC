GitHub
======

How to collaborate to a repository such as `EasyHPC <https://github.com/pescap/EasyHPC>`_? If you want to send Pull Requests to open source directories, you have to follow these steps:

Clone the repository
--------------------

- Run: ::
$ git clone https://github.com/pescap/EasyHPC
$ cd EasyHPC
$ pip install -r requirements.txt

Fork the repository to your GitHub account
------------------------------------------

- Go to the EasyHPC GitHub directory `(click here) <https://github.com/pescap/EasyHPC>`_;
 
- Click on the ``Fork`` button on the top-right hand corner of the window;

- Choose where you want to fork EasyHPC;
  
Work locally on the forked repository
-------------------------------------

Now that you have forked the repository, you will clone it locally on your computer.

- Go to the forked GitHub directory webpage. It should be something like: ::

https://github.com/your_username/EasyHPC/

- Click on the ``clone`` button and copy the directory URL;
  
- In your terminal type::

	$ git clone directory_url

- Go to the directory and create your own branch. For example, assume that you name this branch ``neo``::
  
	$ git branch neo

- Switch to branch ``neo``::
  
  	$ git checkout neo

Now, you are ready to work on this branch, to make all the changes that you want to the code.  

Push to the main repository
---------------------------  	 

Once your changes are done, you can commit and push them to the remote branch: ::

$ git commit -a -m "message about what you added"
$ git push origin neo 

If you want to merge your changes to the original ``EasyHPC``, go to back to your forked page, e.g.: ::

 $ https://github.com/your_username/EasyHPC/

Check compatibility and propose a Pull Request. 

Note: Before you submit a pull request: 

- Verify that your forked version is up to date with the original one;
- Remember to apply `black <https://pypi.org/project/black/>`_ to your Python code. Black allows to format Python code. To install Black and apply it to a ``my_code.py`` script: ::

   	$ pip install black
 	$ black my_code.py

You're ready to collaborate to any Open-Source repository on GitHub! 
