# Git overview

## What is Git?
Git is a distributed version-control system for tracking changes in computer files and coordinating work on those files among multiple people.

## Git terminologies
*  Repository/repo : It is a collection of source codes.
* `commit`: It means to add all modified files to the local repository.
* `branch`: A subsystem of the main branch,master in a git repo(tree) on different machines which work remotely.
* `push`:It means to add all the committed files to the remote repository which can be accessed by all the team members.
* ` merge`:It means to take the independent lines of development created by git branch and integrate them into a single branch.
* `clone`: It means to take an online repo and making a copy of it on the local machine.
* `fork`: Creating a new repo under our name containing all the files from the online repo .

## Git Workflow

![](https://cdn-media-1.freecodecamp.org/images/1*iL2J8k4ygQlg3xriKGimbQ.png)

If you consider a file in your Working Directory, it can be in three possible states:

* It can be staged. Which means the files with the updated changes are marked to be committed to the local repository but not yet committed.
* It can be modified. Which means the files with the updated changes are not yet stored in the local repository.
* It can be committed. Which means that the changes you made to your file are safely stored in the local repository.

## Git Commands

|Command | Description |
| ------ | ------ |
| git init | Initiates an empty git repository |
| git clone [link-to-repository] | To clone the repo |
| git checkout -b [your-branch-name] | To create a new branch |
| git status | Lists all new or modified files to be committed |
| git add . | Adds all the files in the local repository and stages them for commit |
|git commit -m "First commit"|Commit all changes to the local repo|
| git push origin [branch_name]| Push changes to remote repo |
| git pull origin [branch_name]|fetch from remote repo and merge with local repo|
|git reset HEAD~1|remove the most recent commit|


