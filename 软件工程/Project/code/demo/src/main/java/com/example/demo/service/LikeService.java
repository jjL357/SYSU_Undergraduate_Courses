package com.example.demo.service;

import com.example.demo.model.Like;
import com.example.demo.repository.LikeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public interface  LikeService {

   
    public void likePost(Long userId, Long postId);

   
    public void unlikePost(Long userId, Long postId) ;

    public List<Like> findLikesByPostId(Long postId) ;

    public boolean isPostLikedByUser(Long userId, Long postId) ;

    public Long countLikes(Long postId) ;

    // 切换点赞状态
    public boolean toggleLike(Long postId, Long userId) ;
}
