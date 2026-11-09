# ai221_project

## Proposed Project Title:
Machine Learning-Based Classification of Thermal Resistance Degradation in Solder Joints Using Void
Spatial Distribution Features from X-ray Images

## Description of the Problem
In the assembly of semiconductor products, devices are typically soldered onto printed circuit boards
(PCBs) through a reflow soldering process, wherein the device is exposed to a high temperature to melt
the pre-applied solder paste and is then exposed to a lower temperature to let it solidify. During this
process, microscopic gas pockets can become trapped within the solder joint, creating what are known as
solder voids. These voids are especially critical to detect in Class 3 assemblies, which are typically used
in aerospace, automotive, and medical applications, as they can lead to premature failure and reduce
product lifespan.
Solder voids are commonly detected after assembly through non-destructive inspection techniques,
primarily X-ray imaging. Scanning Acoustic Microscopy (SAM) can also be used, mainly for package-
level interfaces or where the board structure allows. Current industry standards and specifications, such
as IPC-A-610, IPC-7095, IPC-7093, and MIL-STD-883, evaluate solder joint and die attach acceptability
primarily based on the total void area, commonly around 25% for solder joints, although the exact
limit varies by component type and customer specification. Some of these documents also address void
type or the size of an individual void, but, to the authors’ knowledge, none defines location-dependent
acceptance limits that relate void position to thermal performance, particularly for thermal pads. This
limitation creates a significant reliability concern. A void occupying 20% of the solder area directly
beneath a thermal pad can obstruct the primary heat dissipation pathway and potentially cause greater
thermal degradation than a similarly sized void distributed near the periphery of the joint. Despite their
substantially different thermal impacts, both conditions are assessed largely on the basis of total void
percentage. Consequently, the relationship between void spatial distribution and thermal performance
remains inadequately addressed by existing acceptance criteria.
This study aims to bridge this gap by investigating whether spatial void distribution characteristics
extracted from X-ray and, where applicable, SAM images can be used to predict thermal resistance
degradation during reliability testing. This study could bring value by potentially giving early insight
into whether a product may fail prematurely, because a released product in manufacturing typically does
not undergo reliability testing. This study could potentially be an added screening test, especially for
Class 3 assemblies. Lastly, it could also potentially reduce the qualification time for new products and

## Contents
- dataset: folder that contains the dataset
- pdf file of project proposal
- jupyter notebook of code
- video presentation
- main paper
- pubmat