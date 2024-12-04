package com.library.booksystem.mapper;

import com.library.booksystem.dto.request.UserRequest;
import com.library.booksystem.dto.response.UserResponse;
import com.library.booksystem.model.User;
import org.mapstruct.Mapper;

@Mapper(componentModel = "spring")
public interface UserMapper {
    UserResponse toUserResponse(User user);
    User toUser(UserRequest request);

}
