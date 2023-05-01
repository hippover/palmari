

<!-- This file is designed to provide you with a starting template for documenting
the functionality of your plugin. Its content will be rendered on your plugin's
napari hub page.

The sections below are given as a guide for the flow of information only, and
are in no way prescriptive. You should feel free to merge, remove, add and 
rename sections at will to make this document work best for your plugin. 

## Description

This should be a detailed description of the context of your plugin and its 
intended purpose.

If you have videos or screenshots of your plugin in action, you should include them
here as well, to make them front and center for new users. 

You should use absolute links to these assets, so that we can easily display them 
on the hub. The easiest way to include a video is to use a GIF, for example hosted
on imgur. You can then reference this GIF as an image.

![Example GIF hosted on Imgur](https://i.imgur.com/A5phCX4.gif)

Note that GIFs larger than 5MB won't be rendered by GitHub - we will however,
render them on the napari hub.

The other alternative, if you prefer to keep a video, is to use GitHub's video
embedding feature.

1. Push your `DESCRIPTION.md` to GitHub on your repository (this can also be done
as part of a Pull Request)
2. Edit `.napari/DESCRIPTION.md` **on GitHub**.
3. Drag and drop your video into its desired location. It will be uploaded and
hosted on GitHub for you, but will not be placed in your repository.
4. We will take the resolved link to the video and render it on the hub.

Here is an example of an mp4 video embedded this way.

https://user-images.githubusercontent.com/17995243/120088305-6c093380-c132-11eb-822d-620e81eb5f0e.mp4

## Intended Audience & Supported Data

This section should describe the target audience for this plugin (any knowledge,
skills and experience required), as well as a description of the types of data
supported by this plugin.

Try to make the data description as explicit as possible, so that users know the
format your plugin expects. This applies both to reader plugins reading file formats
and to function/dock widget plugins accepting layers and/or layer data.
For example, if you know your plugin only works with 3D integer data in "tyx" order,
make sure to mention this.

If you know of researchers, groups or labs using your plugin, or if it has been cited
anywhere, feel free to also include this information here.

## Quickstart

This section should go through step-by-step examples of how your plugin should be used.
Where your plugin provides multiple dock widgets or functions, you should split these
out into separate subsections for easy browsing. Include screenshots and videos
wherever possible to elucidate your descriptions. 

Ideally, this section should start with minimal examples for those who just want a
quick overview of the plugin's functionality, but you should definitely link out to
more complex and in-depth tutorials highlighting any intricacies of your plugin, and
more detailed documentation if you have it.

## Additional Install Steps (uncommon)
We will be providing installation instructions on the hub, which will be sufficient
for the majority of plugins. They will include instructions to pip install, and
to install via napari itself.

Most plugins can be installed out-of-the-box by just specifying the package requirements
over in `setup.cfg`. However, if your plugin has any more complex dependencies, or 
requires any additional preparation before (or after) installation, you should add 
this information here.

## How to Cite

Many plugins may be used in the course of published (or publishable) research, as well as
during conference talks and other public facing events. If you'd like to be cited in
a particular format, or have a DOI you'd like used, you should provide that information here. -->

## Description

Palmari allows you to process your SPT recordings (PALM or other modalities, 2D) with an all-inclusive pipeline: spot detection, sub-pixel localization, tracking & more.
Start quickly with default parameters or customize your pipeline, and run it on entire folder of microscope recordings.

## Quickstart

### On a single recording

To run Palmari on a single microscope recording, click on "Palmari > run Palmari on file..." You'll see a panel open on the right with a pre-loaded default analysis pipeline.

![Default pipeline](https://github.com/hippover/palmari/blob/main/.napari/panel-ouvert.png)

You can run steps of the pipeline one after the other and tweak the parameters so that they suit your experimental setup. Don't forget to set the pixel size and exposure. You can also add and remove processing steps by clicking on the "Edit pipeline" button.

![Visualize results at each step of the process](https://github.com/hippover/palmari/blob/main/.napari/panel-steps.png)

When you're satisfied with te results, just click on "Save locs and tracks" to export localizations and trajectories in a CSV format.

### On a series of files

Once you've set up your processing pipeline, you can save it under the form of a yaml file by clicking on "save pipeline". Then, to use it to process all your acquisitions within a same series of experimental recordings, click on "palmari > run Palmari on folder...", load the pipeline, select the folder where your files lie, and click process !

## What to do next ? Try Tracktor

If you want to test the statistical significance of the difference between the sets of trajectories observed in one or the other experiment (or set of experiments), you may want to try [Tracktor](https://tracktor.pasteur.cloud). It's an online platform developed at tout lab that allows to statistically compare sets of trajectories. 

It notably 
1. estimates the _p_-value of the following null hypothesis "Both these sets of trajectories were generated by the same stochastic process",
2. identifies trajectories that are more found in one set than in another,
3. provides nice visualizations of various standard metrics.

## Documentation

Find more details on Palmari in the [documentation](https://palmari.readthedocs.io/en/latest/).

## Getting Help

Email Hippolyte Verdier : hverdier@pasteur.fr