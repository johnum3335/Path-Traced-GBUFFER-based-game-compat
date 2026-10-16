# Path-Traced-GBUFFER-based-game-compat
RTXPT Renderer based path tracer that hooks on game's g-buffer constants
currently only works on The Last of Us Part II remastered (most of the work) and Silent Hill 2 remake (early work)
Resident evil requiem was used as a guess for how ray reconstruction would denoise the image, DO NOT use it as a renderer on TLOU2
these were vibe-coded with claude.
The Last of us 2 issues

*foliage will flicker alot
*flickering black splotches on characters and other objects
*alot of noise when using certain settings, even with ray reconstruction
*flickering light detection
*culling (im assuming its the issue) will sometimes lit up entire indoor areas for split seconds
*some indoor areas will be completely dark

Silent hill 2 remake issues

*currently WIP
