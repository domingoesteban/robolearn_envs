#!/usr/bin/env bash

echo "Generating urdf files..."

#xacro --inorder target_cylinder.xacro > target_cylinder.urdf
#xacro --inorder cylinder_2d.xacro > cylinder_2d.urdf
xacro --inorder cylinder1_heavy.xacro > cylinder1_heavy.urdf

echo "Done!"
