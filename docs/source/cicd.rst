CI/CD
=====

`EasyHPC <https://github.com/pescap/EasyHPC>`_ supports continuous integration (CI). Here we give a few details concerning how the latter was set. 

Travis CI
---------

`Travis CI <https://www.travis-ci.com/>`_ allows build the package and run unit tests. 
So far, EasyHPC has a Travis check implemented, which just runs a Helloworld function. More tests will be incorporated once some code is added to the git repository.

To add the Travis CI checks to a git repository (e.g. to EasyHPC):

- In your git repository, create a ``.travis.yml`` template (see e.g. `this one <https://github.com/pescap/EasyHPC/blob/main/.travis.yml>`_);
- Create an account on `Travis CI <https://www.travis-ci.com/>`_. It is recommended to link your GitHub account to Travis CI (at the beginning, choose the GitHub option when you sign up for Travis CI.

-  In your `Travis Repositories page <https://app.travis-ci.com/account/repositories>`_, activate the GitHub Apps Integration.


Codacy
------

`Codacy <https://www.codacy.com/>`_ is a very useful tool to produce clean code. To set up your git repository with Codacy:

- Sign up for Codacy (it is recommend to use your GitHub account).
- Add your repository to Codacy

Branch protection rules
-----------------------

To add rules to the pull requests and commits, you can set branch protection rules.

To do so, go to your git repository and click on ``Settings``, and then ``Branches``. Define your custom Branch protection rules.