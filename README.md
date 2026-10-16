# Path-Traced-GBUFFER-based-game-compat
RTXPT Renderer that hooks on game's g-buffer
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
*eyes and hair sometimes glow up under certain conditions, most objects with transparency dont get darkened
<br>
*skin seems to lack subsurface scattering or its original shading is not being properly converted, same applies to hair.
<br>
<br>
Silent hill 2 remake issues
<br>
*currently WIP
<br>
<br>
Install reshade 6.8 or newer with addon support then extract the files to game exe

