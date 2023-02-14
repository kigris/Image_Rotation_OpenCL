__kernel void rotateImage(__read_only image2d_t src_data, __write_only image2d_t dest_data, sampler_t sampler, float sinTheta, float cosTheta, int xAnchor, int yAnchor){
    // Thread id
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    int2 originalCoords={col,row};
    
    // Dimensions sizes
    int cols = get_global_size(0);
    int rows = get_global_size(1);

    // Set default anchor to the middle of the image
    float x0 = cols/2.0f;
    float y0 = rows/2.0f;
    float xOff = col-x0;
    float yOff = row-y0;
    
    // Calculate rotation
    float xPos = ( ((float)(xOff-xAnchor))*cosTheta+((float)(yOff-yAnchor))*sinTheta );
    float yPos = ( ((float)(yOff-yAnchor))*cosTheta-((float)(xOff-xAnchor))*sinTheta);
    
    // Check if the final position is not out of bounds
    if( ( ((int)xPos>=0)&&((int)xPos<cols))
        && ( ((int)yPos>=0)&&((int)yPos<rows)) ){
        int2 coords={(int)xPos,(int)yPos};
        float4 outPixel = read_imagef(src_data, sampler, coords);
        write_imagef(dest_data, originalCoords, outPixel);
    }
}
