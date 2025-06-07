package com.library.booksystem.mapper;

import com.library.booksystem.dto.request.UpdateProfileRequest;
import com.library.booksystem.dto.request.UserRequest;
import com.library.booksystem.dto.response.UserResponse;
import com.library.booksystem.model.User;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;
import org.mapstruct.NullValuePropertyMappingStrategy;

@Mapper(componentModel = "spring", nullValuePropertyMappingStrategy = NullValuePropertyMappingStrategy.IGNORE)
public interface UserMapper {
    UserResponse toUserResponse(User user);
    User toUser(UserRequest request);
    
    void updateUserFromRequest(UpdateProfileRequest request, @MappingTarget User user);
}
