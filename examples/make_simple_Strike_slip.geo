// ---------------------------------------------------------
// 1. PARAMETERS
// ---------------------------------------------------------
SetFactory("OpenCASCADE"); 

res_fault = 4.0;      // Resolution at Top (Surface)
res_depth = 8.0;      // Resolution at Bottom
fault_depth = 20.0;   // Vertical depth of the fault (used for mesh gradient)

// ---------------------------------------------------------
// 2. DEFINE POINTS
// ---------------------------------------------------------
// Top points (z = 0)
Point(1) = {20, 20, 0, res_fault};
Point(2) = {20, 40, 0, res_fault};

// Bottom points (z = -20)
Point(3) = {20, 40, -20, res_depth};
Point(4) = {20, 20, -20, res_depth};

// ---------------------------------------------------------
// 3. DEFINE CURVES (BOUNDARIES)
// ---------------------------------------------------------
Line(1) = {1, 2}; // Top Trace
Line(2) = {2, 3}; // Right Edge
Line(3) = {3, 4}; // Bottom Trace
Line(4) = {4, 1}; // Left Edge

// ---------------------------------------------------------
// 4. GENERATE SURFACE
// ---------------------------------------------------------
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// ---------------------------------------------------------
// 5. PHYSICAL GROUPS
// ---------------------------------------------------------
Physical Surface("Fault_Surface") = {1};
Physical Curve("Fault_Trace_Top") = {1};
Physical Curve("Fault_Trace_Bottom") = {3};

// ---------------------------------------------------------
// 6. FORCE VARIABLE RESOLUTION (Distance Field)
// ---------------------------------------------------------
// Field 1: Calculate Distance from the Top Trace
Field[1] = Distance;
Field[1].CurvesList = {1}; // Curve 1 is the Top Trace
Field[1].Sampling = 100;

// Field 2: Apply Threshold (Linear Gradient)
Field[2] = Threshold;
Field[2].IField = 1;         // Use Distance Field as input
Field[2].LcMin = res_fault;  // Size at DistMin
Field[2].LcMax = res_depth;  // Size at DistMax
Field[2].DistMin = 0.0;      // At the top
Field[2].DistMax = fault_depth; // At the bottom

// Set Background Mesh
Background Field = 2;

// ---------------------------------------------------------
// 7. MESH OPTIONS
// ---------------------------------------------------------
Mesh.Algorithm = 6; // Frontal-Delaunay
Mesh.CharacteristicLengthExtendFromBoundary = 0; 
Mesh.CharacteristicLengthFromPoints = 0; 
Mesh.CharacteristicLengthFromCurvature = 0;