function out = AGCNextFrameOfVector( vLastFrame, vThisInput, aA, aR )

outThisIsLarger = aA * vThisInput + ( 1 - aA ) * vLastFrame;   % attack
outLastIsLarger = aR * vThisInput + ( 1 - aR ) * vLastFrame;   % release

out = ( vThisInput > vLastFrame ) .* outThisIsLarger + ( vThisInput <= vLastFrame ) .* outLastIsLarger;