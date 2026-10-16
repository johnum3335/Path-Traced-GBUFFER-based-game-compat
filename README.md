# Path-Traced-GBUFFER-based-game-compat
RTXPT Renderer based path tracer that hooks on game's g-buffer constants.
<br>
<br>
currently only works on The Last of Us Part II remastered (most of the work) and Silent Hill 2 remake (early work)
Resident evil requiem was used as a guess for how ray reconstruction would denoise the image. <br> DO NOT use it as a renderer on TLOU2
<br>
vibe-coded with claude.
<br>
<br>
The Last of us 2 issues

*foliage will flicker alot
<br>
*flickering black splotches on characters and other objects
<br>
*alot of noise when using certain settings, even with ray reconstruction
<br>
*flickering light detection
<br>
*culling (im assuming its the issue) will sometimes lit up entire indoor areas for split seconds
<br>
*some indoor areas will be completely dark
<br>
*eyes sometimes glow up under certain conditions, most objects with transparency dont get darkened
<br>
<br>
Silent hill 2 remake issues
<br>
*currently WIP
