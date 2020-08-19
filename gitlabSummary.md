# Gitlab
GitLab is a 2nd most popular remote storage solution for Git-based repository and a powerful complete application for software development with an "user-and-newbie-friendly" interface.


# GitLab WorkFlow
With the GitLab Workflow, the goal is to help teams work cohesively and effectively from the first stage of implementing something new (ideation) to the last stage—deploying implementation to production. 
<br>Here are the 10 stages:
<img src="https://about.gitlab.com/images/blogimages/idea-to-production-10-steps.png" width="80%">

## Creating an issue
* It is mainly used for *Discussing ideas, Asking questions, Reporting bugs and malfunction.*<br>
* Navigate to your Project’s Dashboard > Issues > List > New Issue
<img src="https://docs.gitlab.com/ee/user/project/issues/img/new_issue_from_tracker_list.png" width="70%">

### First Commit
`git commit -m "this is my commit message. Ref #xxx"`<br>
> xxx is the issue number which is used to create a link between the issue and the first commit related to it.


##  Making a Merge Request
A Merge Request is a basis of GitLab as a code collaboration and version control.<br>
Once new changes are pushed to a new branch to GitLab, on visiting the repository we can see a call-to-action at the top of your screen from<br>
which you can click the button **Create Merge Request**.
<img src="https://docs.gitlab.com/ee/user/project/merge_requests/img/create_merge_request_button_v12_6.png" width="70%">
<br>
**or**<br><br>
Navigate to Project page> Repository > Files page > Left Nav bar "Merge Requests" tab > **New Merge Request** button.<br><br>
<img src="extras/newimg.PNG" width="80%">


## Getting Feedback and Merging
* After creating a merge request, it's time to get feedback from your team or collaborators. Using the diffs available on the UI, we can add inline comments, reply to them and resolve them.
<img src="https://about.gitlab.com/images/blogimages/gitlab-workflow-an-overview/gitlab-code-review.png" width="70%"><br>
* The assignee can see the merge request and can approve or close the request.<br>
<img src="extras/merge_req.JPG" width="50%">

## Milestones
Milestones in GitLab are a way to track issues and merge requests created to achieve a broader goal in a certain period of time.