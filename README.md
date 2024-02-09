# About Meridian

Marketing mix modeling (MMM) is a statistical analysis technique that measures
the impact of marketing campaigns and activities to guide budget planning
decisions and improve overall media effectiveness. MMM uses aggregated data to
measure impact across marketing channels and account for non-marketing factors
that impact sales and other key performance indicators (KPIs). MMM is
privacy-safe and does not use any cookie or user-level information.

Meridian is an MMM framework that enables advertisers to set up and run their
own in-house models. Meridian helps you answer key questions such as:

*   How did the marketing channels drive my sales (or other KPI)?
*   What was my marketing return on investment (ROI)?
*   How do I optimize my marketing budget allocation for the future?

Meridian is a highly customizable modeling framework that is based on
[Bayesian causal inference](https://developers.google.com/meridian/docs/basics/bayesian-inference).
It is capable of handling large scale geo-level data, which is encouraged if
available, but it can also be used for national-level modeling. Meridian
provides clear insights and visualizations to inform business decisions around
marketing budget and planning. Additionally, Meridian provides methodologies to
support calibration of MMM with experiments and other prior information, and to
optimize target ad frequency by utilizing reach and frequency data.

If you are using LightweightMMM, see the
[migration guide](https://developers.google.com/meridian/docs/migrate) to help
you understand the differences between these MMM products.

# Install Meridian

#### During Limited Beta Launch

Run the following command to automatically install the most recent version from
GitHub:

```sh
$ git clone https://github.com/google/meridian/google_mmm.git
$ pip install .
```

#### During Post GA Launch

```sh
$ pip install --upgrade google-meridian
```

## How to use the Meridian library

To get started with Meridian, you can run the code programmatically using sample
data with the
[Getting Started Colab](https://colab.corp.google.com/drive/1rl4bZu4fXRqkVrFWmvAH6TeLxMPXU12q).

The Meridian model uses a holistic MCMC sampling approach called
[No U Turn Sampler (NUTS)](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/NoUTurnSampler)
which can be compute intensive. To help with this, GPU support has been
developed across the library (out-of-the-box) using tensors. We recommend
running your Meridian model on GPUs to get real time optimization results and
significantly reduce training time.

# Meridian Documentation & Tutorials

The following documentation, colab, and video resources will help you get
started quickly with using Meridian:

Resource                                                                                                                        | Description
------------------------------------------------------------------------------------------------------------------------------- | -----------
[Meridian documentation](https://developers.google.com/meridian)                                                                | Main landing page for Meridian documentation.
[Meridian basics](https://developers.google.com/meridian/docs/basics/about-the-product)                                         | Learn about Meridian features, methodologies, and the model math.
[Getting started colab](https://colab.google.com/drive/1rl4bZu4fXRqkVrFWmvAH6TeLxMPXU12q)                                  | Install and quickly learn how to use Meridian with this colab tutorial using sample data.
[User guide](https://developers.google.com/meridian/docs/user-guide/overview)                                                   | A detailed walk-through of how to use Meridian and generating visualizations using your own data.
[Advanced modeling considerations](https://developers.google.com/meridian/docs/advanced-modeling/model-fit)        | Advanced modeling guidance for model refinement and edge cases.
[Model debugging and troubleshooting](https://developers.google.com/meridian/docs/model-debugging/model-debugging) | Guidance for coding errors, model debugging, and troubleshooting.
[Migrate from LMMM](https://developers.google.com/meridian/docs/migrate)                                                        | Learn about the differences between Meridian and LightweightMMM as you consider migrating.
[API Reference](https://developers.google.com/meridian/docs/api)                                                                | API reference documentation for the Meridian package.
[Reference list](https://developers.google.com/meridian/docs/reference-list)                                                    | White papers and other referenced material.
