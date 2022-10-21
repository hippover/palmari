.. palmari documentation main file, created by
   sphinx-quickstart on Thu Oct  1 00:43:18 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Palmari's documentation!
===================================

What's included ?
-----------------

Palmari is a Napari plugin providing tools to process movies of PALM experiments (photo-activated localization microscopy). 
It provides a **customizable pipeline scheme** with the following features :

- **Get your trajectories in a few clicks** with built-in, ready-to-use processing steps : localizer, tracker, drift corrector, ... 

- **Direct visualization of results** to intuitively adjust a pipeline's parameters : visualize the effect of one or the other parameter. 
  Want to test your results' robustness to different processing pipelines ? 
  Palmari stores the localizations and trajectories output from each pipeline so that they can be easily compared afterwards.

- Designed from the start to **process series of PALM acquisitions** using a same pipeline : 
  don't loose time writing scripts to go through all files in a folder. 
  Once your pipeline is ready, process batches of experiments in a few clicks, direclty from Napari.

- Want to take advantage of HPC infrastructure ? Palmari has a Python interface.
  Its :py:class:`Experiment` class allows you to keep track of your entire series of acquisitions and stores your processing results.  
  Processing steps rely on Dask, so as to take advantage of multithreading when several cores are available.

- Easily **include your favorite processing steps** in your Palmari pipeline, see :ref:`here <own_steps>` for more details on implementations. 
  If it's worth sharing, consider a merge request ! 🫵 

.. figure:: images/plugin_steps.png

   Visualize processing steps and tweak their parameters with Palmari's interface in Napari.

.. important::

   This package is under development, if you wish to contribute, report a bug or suggest an addition, raise an issue or send an email to hverdier@pasteur.fr.
   Contributions enriching the set of built-in image processing steps, among others, are very welcome !


Installation
-------------

Install Palmari either via the Napari plugin manager, or using ``pip``:

.. code-block::

   pip install palmari

Contents
---------

.. toctree::
   :maxdepth: 3

   Napari plugin <napari>
   TIF processing pipeline <pipeline>
   Data structure <data>
   Examples <examples>
   Authors <authors>
   API index <api/modules>
