package com.example.demo.repository;

import org.springframework.data.repository.CrudRepository;
import com.example.demo.model.Post;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PostRepository extends JpaRepository<Post, Integer> {
    List<Post> findByAuthorId(Long authorId);
    Post findByPostId(Integer postId);
    
}
