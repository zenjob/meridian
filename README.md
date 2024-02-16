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
you understand the differences between these MMM projects.

# Install Meridian

Run the following command to automatically install the most recent version from
GitHub:

```sh
$ git clone https://github.com/google/meridian.git
$ pip install .
```

## How to use the Meridian library

To get started with Meridian, you can run the code programmatically using sample
data with the [Getting Started Colab][3].

The Meridian model uses a holistic MCMC sampling approach called
[No U Turn Sampler (NUTS)](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/mcmc/NoUTurnSampler)
which can be compute intensive. To help with this, GPU support has been
developed across the library (out-of-the-box) using tensors. We recommend
running your Meridian model on GPUs to get real time optimization results and
significantly reduce training time.

# Meridian Documentation & Tutorials

The following documentation, colab, and video resources will help you get
started quickly with using Meridian:

| Resource                    | Description                                    |
| --------------------------- | ---------------------------------------------- |
| [Meridian documentation][1] | Main landing page for Meridian documentation.  |
| [Meridian basics][2]        | Learn about Meridian features, methodologies, and the model math. |
| [Getting started colab][3]  | Install and quickly learn how to use Meridian with this colab tutorial using sample data. |
| [User guide][4]             | A detailed walk-through of how to use Meridian and generating visualizations using your own data. |
| [Advanced modeling considerations][5]    | Advanced modeling guidance for model refinement and edge cases. |
| [Model debugging and troubleshooting][6] | Guidance for coding errors, model debugging, and troubleshooting. |
| [Migrate from LMMM][7]      | Learn about the differences between Meridian and LightweightMMM as you consider migrating. |
| [API Reference][8]          | API reference documentation for the Meridian package. |
| [Reference list][9]         | White papers and other referenced material.    |

[1]: https://developers.google.com/meridian
[2]: https://developers.google.com/meridian/docs/basics/about-the-project
 <!-- TODO: this colab link is a placeholder -->
[3]: https://colab.research.google.com/
[4]: https://developers.google.com/meridian/docs/user-guide/overview
[5]: https://developers.google.com/meridian/docs/advanced-modeling/model-fit
[6]: https://developers.google.com/meridian/docs/model-debugging/model-debugging
[7]: https://developers.google.com/meridian/docs/migrate
[8]: https://developers.google.com/meridian/docs/api
[9]: https://developers.google.com/meridian/docs/reference-list
