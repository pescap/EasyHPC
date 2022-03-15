CI/CD
=====

`EasyHPC <https://github.com/pescap/EasyHPC>`_ supports continuous integration (CI). Here we give a few details concerning how the latter was set. 


reStructuredText (RST)
----------------------

The reStructuredText (RST) syntax provides an easy-to-read, what-you-see-is-what-you-get plaintext markup syntax and parser system. However, you need to be very precise and stick to some strict rules:

- like Python, RST syntax is sensitive to indentation !
- RST requires blank lines between paragraphs


This entire page is written with the RST syntax. In the landing page, you should find a link to the repository, which shows the RST source code.

**Text Formating: inline markup and special characters (e.g., bold, italic, verbatim)**




There are a few special characters used to format text. The special character ``*`` is used to defined bold and italic text as shown in the table below. The backquote character ````` is another special character used to create links to internal or external web pages as you will see in section `Internal and External Links`_.

=========== ================================== ==============================
usage          syntax                           HTML rendering
=========== ================================== ==============================
italic      `*italic*`                         *italic*
bold        `**bold**`                         **bold**
link        ```python <www.python.org>`_``     `python <www.python.org>`_
verbatim    ````*````                               ``*``
=========== ================================== ==============================

**Headings**

In order to write a title, you can either underline it or under and overline it. The following examples are correct titles.

.. code-block:: rest

    *****
    Title
    *****

    subtitle
    ########

    subsubtitle
    **********************
    and so on





Two rules: 

  * If under and overline are used, their length must be identical
  * The length of the underline must be at least as long as the title itself

Normally, there are no heading levels assigned to certain characters as the 
structure is determined from the succession of headings. However, it is better to stick to the same convention throughout a project. For instance: 

* `#` with overline, for parts
* `*` with overline, for chapters
* `=`, for sections
* `-`, for subsections
* `^`, for subsubsections
* `"`, for paragraphs

**Internal and External Links**


In Sphinx, you have 3 type of links:
    #. External links (http-like)
    #. Implicit links to title
    #. Explicit links to user-defined label (e.g., to refer to external titles).


**External links**


If you want to create a link to a website, the syntax is ::

    `<http://www.python.org/>`_

which appear as `<http://www.python.org/>`_ . Note the underscore after the final single quote. Since the full name of the link is not always simple or meaningful, you can specify a label (note the space between the label and link name)::

    `Python <http://www.python.org/>`_

The rendering is now: `Python <http://www.python.org/>`_. 

.. note:: If you have an underscore within the label/name, you got to escape it with a '\\' character.


.. _implicit:

**Implicit Links to Titles**


All titles are considered as hyperlinks. A link to a title is just its name within quotes and a final underscore::

    `Codacy`_

This syntax works only if the title and link are within the same RST file.
If this is not the case, then you need to create a label before the title and refer to this new link explicitly.

**List and bullets**


The following code::

    * This is a bulleted list.
    * It has two items, the second
      item uses two lines. (note the indentation)

    1. This is a numbered list.
    2. It has two items too.

    #. This is a numbered list.
    #. It has two items too.

gives:

* This is a bulleted list.
* It has two items, the second
  item uses two lines. (note the indentation)

1. This is a numbered list.
2. It has two items too.

#. This is a numbered list.
#. It has two items too.

.. note:: if two lists are separated by a blanck line only, then the two lists are not differentiated as you can see above.

If you want to learn more about .rst files just visit `Here <https://thomas-cokelaer.info/tutorials/sphinx/index.html>`_ 



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