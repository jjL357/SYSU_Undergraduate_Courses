package com.example.demo.service.impl;

import com.example.demo.model.User;
import com.example.demo.model.Post;
import com.example.demo.repository.LikeRepository;
import com.example.demo.repository.PostRepository;
import com.example.demo.repository.UserRepository;
import com.example.demo.service.UserService;
import com.example.demo.service.PostService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.ModelAttribute;
import com.example.demo.dao.LikeDAO;
import java.util.ArrayList;
import java.util.List;

@Service
public class PostServiceImpl implements PostService {

    @Autowired
    private PostRepository postRepository;
    
    @Autowired
    private LikeDAO likeDAO;

    @Autowired
    private LikeRepository likeRepository;

    public Post savePost(@ModelAttribute Post post) {
        return postRepository.save(post);
    }

    public List<Post> getPostsByAuthorId(Long authorId) {
        return postRepository.findByAuthorId(authorId);
    }

    public void deletePost(Integer postId) {
        postRepository.deleteById(postId);
    }

    public Post getPostByPostId(Integer postId) {
        return postRepository.findByPostId(postId);
    }
    
    public List<Post> getAllPosts() {
        return postRepository.findAll();
    }

    @Override
    public Long getLikeCount(Integer postId) {
        return likeRepository.countByPostId(postId);
    }

    public List<Post> findHotPosts() {
        List<Object []> hotPostsWithLikes = likeDAO.findTop15HotPosts();
        // Transforming the result into Post objects
        List<Post> hotPosts = new ArrayList<>();
        for (Object [] result : hotPostsWithLikes) {
            Long postid = (Long)result[0];
            Post post =  postRepository.findByPostId(postid.intValue());
            hotPosts.add(post);
        }
        return hotPosts;
    }
}
