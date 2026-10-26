============
Contributing
============

EasyHPC grows with the questions of the people who use it. If something was hard to find, wrong or missing, it is worth fixing here for the next person.


Report a problem
----------------

Open an `issue <https://github.com/pescap/EasyHPC/issues>`__ with the page, what you tried and what happened. A copy of the command and the error message is the most useful thing you can include.


Propose a change
----------------

#. Fork the repository and create a branch, as explained in :doc:`github`.
#. Edit or add an RST file in ``docs/source``. A new page also needs an entry in one of the ``toctree`` blocks of ``index.rst``.
#. Build the documentation locally and read your page in the browser (see :doc:`cicd`).
#. Send a pull request. GitHub Actions builds the documentation and reports any error.

Every page also has an edit button at the top, which opens the source file on GitHub.


Style
-----

- One page per topic, with a short introduction that says what the reader will be able to do.
- Prefix shell commands with ``$``. The copy button skips the prompt, so readers can paste the commands as they are.
- Use placeholders such as ``user``, ``cluster`` or ``my_script.py``, and never real passwords, tokens, private links or internal host names.
- Link to the official documentation for details, and keep here the commands that work.
- Use only images you created or that have a license that allows reuse, and cite their source.
