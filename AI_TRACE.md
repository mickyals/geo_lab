# Project AI Trace

## Jeb
I primarily used AI to assist with adapting and improving our visualization code throughout the project. I created the code myself, which then proved to be sufficient for Claude to base further adaptations on the existing plots. In general, these were relatively limited tasks that Claude could successfully one-shot given the context, and the fact that the code output visualizations made it easier to verify the output’s correctness by eye. I will note mistakes and corrections under each chat.

**Adding Colorcet** [Link](https://claude.ai/share/34f53a98-eaa0-4fd9-ab2d-623ef99a5c2e)
Mickell asked about colorcet for our colourmaps, and I was not familiar. I discussed suggestions for colourmaps to choose from the colorcet selection (we ended up choosing different ones by hand). I submitted the colorcet getting started docs to ground Claude’s usage.

**Adding Wind Magnitude** [Link](https://claude.ai/share/5b3fbd73-cc4f-4173-a167-5a8b193d54d2)
Claude adapted the existing variable plots to create a new synthetic wind magnitude plot using zonal and meridional wind (u and v). We had previously been caught out by failing to denormalize our values for plotting, so I doublechecked Claude’s output on this key factor. 

**Creating Ground Truth Plots** [Link](https://claude.ai/share/48af888a-a372-48d5-ae3c-c1f68f08aa27)
I got Claude to adapt our existing visualizer code to create a script that would load the original data and create ground truth plots for comparison. Usually AI is good at these transference/translation tasks, but here it got the plots backwards and upside down at first, but was able to fix it once I pointed out the errors.

**Fixing Values Ranges for Plots** [Link](https://claude.ai/share/f262540c-7c47-49e3-90f6-9ce15ee98694)
Claude helped me analyze the ground truth data to determine the appropriate min and max values to make our plots comparable - otherwise matplotlib would select different values that would render the colours difficult to compare directly between plots. 

**Expanding Grouth Truth Plotting** [Link](https://claude.ai/share/ae5204ba-305c-4ea6-b8f1-5717a72a3590)
Here we expanded the ground truth plotting script to also include two different vertical slices of the atmosphere. We had already been generating these from the model. This exercise actually revealed that I had made a serious mistake when I originally created the vertical slice visualization code: we depict our horizontal slices at three different heights (represented as pressure levels), and I had unthinkingly only sampled from these three pressure levels for the vertical slices rather than the full 17 from the data! Matplotlib smoothly interpolated between the samples, so it was not obvious that we were undersampling in the vertical slices - we assumed the models were just struggling with vertical structure. Interestingly, it was Claude mentioning how it had implemented the reduced sampling in its adaptation that made me realize the error.