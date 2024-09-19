package com.example.demo.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.ModelAttribute;

import com.example.demo.model.Post;
import com.example.demo.repository.PostRepository;

import java.util.List;


public interface  PostService {

  

    public Post savePost(@ModelAttribute Post post);

    public List<Post> getPostsByAuthorId(Long authorId) ;

    public void deletePost(Integer postId);

    public Post getPostByPostId(Integer postid);
    public List<Post> getAllPosts();
    
}


