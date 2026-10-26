=====
CI/CD
=====

`EasyHPC <https://github.com/pescap/EasyHPC>`__ is written in reStructuredText, built with Sphinx, checked by GitHub Actions on every pull request and published on Read the Docs. This page explains each piece so you can reuse the setup in your own projects.


reStructuredText (RST)
----------------------

reStructuredText is a plain-text markup language. It is easy to read, but it follows a few strict rules:

- Like Python, RST is sensitive to indentation.
- Paragraphs, lists and code blocks must be separated by blank lines.

Every page of this documentation is an RST file in `docs/source <https://github.com/pescap/EasyHPC/tree/main/docs/source>`__. Reading them next to the rendered pages is the fastest way to learn the syntax.


Text formatting
***************

The source

.. code-block:: rst

    *italic*, **bold**, ``literal text`` and a link to `Python <https://www.python.org>`__

renders as: *italic*, **bold**, ``literal text`` and a link to `Python <https://www.python.org>`__.


Headings
********

A heading is a line of text underlined (and optionally overlined) with a punctuation character:

.. code-block:: rst

    ==========
    Page title
    ==========

    Section
    -------

    Subsection
    **********

Two rules:

- If you use both an overline and an underline, they must have the same length.
- The underline must be at least as long as the title.

RST does not assign a level to each character: the level is given by the order in which the characters first appear in the file. Stick to one convention across the project. EasyHPC uses ``=`` above and below the page title, ``-`` for sections and ``*`` for subsections.


Code
****

End a paragraph with ``::`` and indent the next block to display it as code::

    $ echo "Hello, cluster"

To choose the highlighting language, use the ``code-block`` directive:

.. code-block:: rst

    .. code-block:: python

        print("Hello, cluster")


Links
*****

External links:

.. code-block:: rst

    `Python <https://www.python.org>`__

Links to another page of the documentation, using the file name without extension:

.. code-block:: rst

    :doc:`sysadmin`

Links to a section. Add a label before the section title, then refer to it from any page:

.. code-block:: rst

    .. _my-label:

    My section
    ----------

    See :ref:`my-label`.


Lists
*****

The source

.. code-block:: rst

    * This is a bulleted list.
    * It has two items. The second item
      spans two lines (note the indentation).

    #. This is a numbered list.
    #. Sphinx numbers the items for you.

renders as:

* This is a bulleted list.
* It has two items. The second item
  spans two lines (note the indentation).

#. This is a numbered list.
#. Sphinx numbers the items for you.

To learn more, see the `reStructuredText primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__ in the Sphinx documentation.


Build the documentation locally
-------------------------------

Before sending a pull request, build the pages on your computer and read them in a browser::

    $ pip install -r requirements.txt
    $ sphinx-build -b html docs/source docs/build/html
    $ open docs/build/html/index.html

On Linux, replace ``open`` with ``xdg-open``. Sphinx prints a warning for every syntax error, with the file and line number.


GitHub Actions
--------------

`GitHub Actions <https://docs.github.com/en/actions>`__ runs workflows on GitHub's machines. Workflows are YAML files stored in `.github/workflows <https://github.com/pescap/EasyHPC/tree/main/.github/workflows>`__.

EasyHPC's `docs.yml <https://github.com/pescap/EasyHPC/blob/main/.github/workflows/docs.yml>`__ builds the documentation on every pull request and on every push to ``main``:

.. code-block:: yaml

    name: Docs

    on:
      push:
        branches: [main]
      pull_request:

    jobs:
      build:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: "3.12"
          - run: pip install -r requirements.txt
          - run: sphinx-build --keep-going -b html docs/source docs/build/html

If the build fails, the pull request shows a red cross and the log of the failing step.


Read the Docs
-------------

`Read the Docs <https://readthedocs.org>`__ builds and hosts the documentation at `easyhpc.readthedocs.io <https://easyhpc.readthedocs.io>`__. It rebuilds the site after every push to ``main``.

Read the Docs requires a ``.readthedocs.yaml`` file at the root of the repository:

.. code-block:: yaml

    version: 2

    build:
      os: ubuntu-24.04
      tools:
        python: "3.12"

    sphinx:
      configuration: docs/source/conf.py

    python:
      install:
        - requirements: requirements.txt

To publish a project of your own, sign in to Read the Docs with your GitHub account and import the repository.


Codacy
------

`Codacy <https://www.codacy.com/>`__ analyzes the code of a repository and reports style issues and possible bugs. To set it up:

- Sign up for Codacy with your GitHub account.
- Add your repository.


Branch protection rules
-----------------------

Branch protection rules prevent direct pushes to ``main`` and can require a passing build and an approved review before merging.

To set them up, go to your repository on GitHub, click on ``Settings`` and then ``Branches``, and add a rule for ``main``.
