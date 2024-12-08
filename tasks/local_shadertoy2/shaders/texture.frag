#version 430
#extension GL_GOOGLE_include_directive : require


#include "cpp_glsl_compat.h"


layout(location = 0) out vec4 out_fragColor;

layout(push_constant) uniform params_t
{
  uvec2 resolution;
} params;


// code stolen from here: https://www.shadertoy.com/view/lsf3WH

// 0: integer hash
// 1: float hash (aliasing based) (don't do this unless you live in the year 2013)
#define METHOD 0

// 0: cubic
// 1: quintic
#define INTERPOLANT 0

#if METHOD==0
float hash( in ivec2 p )  // this hash is not production ready, please
{                         // replace this by something better

    // 2D -> 1D
    int n = p.x*3 + p.y*113;

    // 1D hash by Hugo Elias
	n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0+2.0*float( n & 0x0fffffff)/float(0x0fffffff);
}
#else
float hash(vec2 p)  // replace this by something better
{
    p  = 50.0*fract( p*0.3183099 + vec2(0.71,0.113));
    return -1.0+2.0*fract( p.x*p.y*(p.x+p.y) );
}
#endif

float noise( in vec2 p )
{
    #if METHOD==0
    ivec2 i = ivec2(floor( p ));
    #else
    vec2 i = floor( p );
    #endif
    vec2 f = fract( p );
	
    #if INTERPOLANT==1
    // quintic interpolant
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    #else
    // cubic interpolant
    vec2 u = f*f*(3.0-2.0*f);
    #endif    

    #if METHOD==0
    return mix( mix( hash( i + ivec2(0,0) ), 
                     hash( i + ivec2(1,0) ), u.x),
                mix( hash( i + ivec2(0,1) ), 
                     hash( i + ivec2(1,1) ), u.x), u.y);
    #else
    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
    #endif
}


void main() {
  vec2 uv = gl_FragCoord.xy / params.resolution;
  
  float f = 0.0;   
  uv *= 8.0;
      mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
  	f  = 0.5000*noise( uv ); uv = m*uv;
  	f += 0.2500*noise( uv ); uv = m*uv;
  	f += 0.1250*noise( uv ); uv = m*uv;
  	f += 0.0625*noise( uv ); uv = m*uv;  
  f = 0.5 + 0.5*f;

  out_fragColor = vec4( f*0.6, f*0.6, 0., 1.0 );
}
