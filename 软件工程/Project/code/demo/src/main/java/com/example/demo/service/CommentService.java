package com.example.demo.service;

import com.example.demo.model.Comment;
import com.example.demo.repository.CommentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class CommentService {

    private final CommentRepository commentRepository;

    @Autowired
    public CommentService(CommentRepository commentRepository) {
        this.commentRepository = commentRepository;
    }

    public List<Comment> getAllCommentsByPostId(Long postId) {
        return commentRepository.findByPostId(postId);
    }

    public Comment getCommentById(Long commentId) {
        Optional<Comment> commentOptional = commentRepository.findById(commentId);
        return commentOptional.orElse(null); // 如果不存在，则返回 null
    }

    public Comment saveComment(Comment comment) {
        return commentRepository.save(comment);
    }

    public Comment replyToComment(Long postId, Long parentId, String content) {
        Comment parentComment = commentRepository.findById(parentId).orElse(null);
        if (parentComment == null) {
            // 处理父评论不存在的情况，可以抛出异常或者返回 null 或者自定义的错误处理
            throw new IllegalArgumentException("Parent comment with id " + parentId + " not found.");
        }

        Comment reply = new Comment();
        reply.setPostId(postId);
        reply.setParentId(parentId);
        reply.setContent(content);

        // 保存回复评论，并将回复添加到父评论的子评论列表中
        Comment savedReply = commentRepository.save(reply);
        // 更新父评论
        commentRepository.save(parentComment);

        return savedReply;
    }

    public void deleteComment(Long commentId) {
        commentRepository.deleteById(commentId);
    }
}
