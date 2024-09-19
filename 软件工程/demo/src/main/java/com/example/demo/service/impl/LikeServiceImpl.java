package com.example.demo.service.impl;

import com.example.demo.dao.LikeDAO;
import com.example.demo.model.User;
import com.example.demo.model.Post;
import com.example.demo.model.Like;
import com.example.demo.repository.PostRepository;
import com.example.demo.repository.UserRepository;
import com.example.demo.repository.LikeRepository;
import com.example.demo.service.UserService;
import com.example.demo.service.PostService;
import com.example.demo.service.LikeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.ModelAttribute;

import java.util.List;

@Service
public class LikeServiceImpl implements LikeService {
    @Autowired
    private LikeRepository likeRepository;


    @Autowired
    private LikeDAO likeDAO;

   
    public void likePost(Long userId, Long postId) {
        Like like = new Like(userId, postId);
        likeRepository.save(like);
    }

   
    public void unlikePost(Long userId, Long postId) {
        Like like = likeRepository.findByUserIdAndPostId(userId, postId);
        if (like != null) {
            likeRepository.delete(like);
        }
    }

    public List<Like> findLikesByPostId(Long postId) {
        return likeRepository.findByPostId(postId);
    }

    public boolean isPostLikedByUser(Long userId, Long postId) {
        Like like = likeRepository.findByUserIdAndPostId(userId, postId);
        return like != null;
    }

    public Long countLikes(Long postId) {
        return likeDAO.countLikes(postId);
    }

    // 切换点赞状态
    public boolean toggleLike(Long postId, Long userId) {
        Like like = likeRepository.findByPostIdAndUserId(postId, userId);
        if (like == null) {
            // 如果用户未点赞，创建点赞记录
            like = new Like();
            like.setPostId(postId);
            like.setUserId(userId);
            likeRepository.save(like);
            return true; // 点赞成功
        } else {
            // 如果用户已经点赞，取消点赞
            likeRepository.delete(like);
            return false; // 取消点赞成功
        }
    }
}
