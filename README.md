# Automating Some Elements of Paper Marking Sheets. 

## Stamp Aruco Corners 
```
python .\stamp_aruco_corners.py .\MarkingSheet.pdf Alignable_Marking_Sheet.pdf --marker-mm 4 --inset-mm 5 --dict 4X4_50
```

This script adds markers to the corners of the marking sheet that are small enough to not cause issues, but allow for the marking sheets to be merged following the sheets being filled. This is useful for TA teams where one TA may mark a random subset of students, but all marks need to be entered.

## Align and Merge
```
python .\align_and_merge.py .\SessionScans.pdf .\Merged.pdf
```

This script takes a scan of all the pages from all the TAs and automatically identifies the Markers on the corners using them to align the sheets together and reassemble a PDF with the different sheets merged together. This one marking sheet is then substantially faster to enter than going through the sheets independently.

## GUI (Qt)
```
python gui.py
```

Provides tabs for both stamping ArUco corners and aligning/merging marked PDFs. Requires PyQt5 in addition to the existing dependencies.

## Setup
Install dependencies with:
```
pip install -r requirements.txt
```
