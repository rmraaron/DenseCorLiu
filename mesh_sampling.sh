#!/bin/bash

FILES="./subjects/*.obj"

for file in $FILES
do
	filename=`echo "$file" | sed -e 's/\(.obj\)*$//g'`
	filename=`echo "$filename" | sed -e 's/subjects/mesh_sampling/g'`
	fn="${filename}.pcd"
	pcl_mesh_sampling $file $fn -n_samples 2048 -leaf_size 0.005 -no_vis_result
	echo $fn
done


