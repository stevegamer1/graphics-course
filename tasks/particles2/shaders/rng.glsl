uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

uint rng_state = 0;
uint rng(uint time, uint particle_index) {
    rng_state = hash(uvec4(time, particle_index, rng_state, 0));
    return hash(uvec4(time, particle_index, rng_state, 1));
}